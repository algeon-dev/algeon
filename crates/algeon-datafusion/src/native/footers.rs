//! Parquet footers shared by the source-lowering steps of one planning run.
//!
//! One lowering reads the footer of a scan object from several places (schema
//! mapping, byte-range coalescing, row-group pruning, byte metadata, source
//! diagnostics), and one optimizer run lowers the same plan more than once
//! (whole plan, mixed leaf, GPU relation inputs, graph residency). A
//! [`FooterScope`] entered on the planning thread lets every read in that
//! window share one parse per local file. Reads outside a scope parse
//! independently.
//!
//! A remote object's footer comes from the planning session's DataFusion file
//! metadata cache, which a `ListingTable` or table provider fills during its
//! async `scan`. Physical optimizer rules run synchronously on the caller's
//! Tokio worker, so planning never reads a remote footer from the store; an
//! object missing from that cache has no footer here.

#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{
    cell::RefCell,
    collections::HashMap,
    fs::{self, File},
    marker::PhantomData,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, PoisonError},
    time::SystemTime,
};

use datafusion::{
    datasource::physical_plan::parquet::metadata::CachedParquetMetaData,
    execution::{cache::cache_manager::FileMetadataCache, object_store::ObjectStoreUrl},
    object_store::ObjectMeta,
};
use parquet::{
    errors::Result as ParquetResult,
    file::metadata::{ParquetMetaData, ParquetMetaDataReader},
};

use super::pruning::local_object_path_to_filesystem;

/// Footer bytes one scope retains. Every source step walks all files of a
/// scan, so an unbounded cache would hold the footer of every file in a wide
/// table at once. Past the budget a footer is parsed on each request.
const FOOTER_CACHE_BUDGET_BYTES: usize = 64 * 1024 * 1024;

thread_local! {
    static ACTIVE_CACHE: RefCell<Option<Arc<FooterCache>>> = const { RefCell::new(None) };
}

#[cfg(test)]
thread_local! {
    static THREAD_PARSES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Number of footers parsed on this thread, cached or not.
#[cfg(test)]
pub(crate) fn footer_parses_on_this_thread() -> usize {
    THREAD_PARSES.get()
}

/// Footers keyed by path, size, and modification time, so a file rewritten
/// during the scope misses the cache and is parsed again.
pub(crate) struct FooterCache {
    state: Mutex<CacheState>,
    budget_bytes: usize,
    /// The planning session's DataFusion file metadata cache, which serves
    /// remote footers.
    remote: Option<Arc<FileMetadataCache>>,
    #[cfg(test)]
    parses: AtomicUsize,
}

impl std::fmt::Debug for FooterCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FooterCache")
            .field("budget_bytes", &self.budget_bytes)
            .field("remote", &self.remote.is_some())
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Default)]
struct CacheState {
    footers: HashMap<FooterKey, Arc<ParquetMetaData>>,
    retained_bytes: usize,
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct FooterKey {
    path: PathBuf,
    size: u64,
    modified: SystemTime,
}

impl FooterCache {
    /// A cache with the budget every planning scope retains. `remote` is the
    /// planning session's file metadata cache; without it, remote objects
    /// have no footer.
    pub(crate) fn for_scope(remote: Option<Arc<FileMetadataCache>>) -> Self {
        Self { remote, ..Self::new(FOOTER_CACHE_BUDGET_BYTES) }
    }

    fn new(budget_bytes: usize) -> Self {
        Self {
            state: Mutex::new(CacheState::default()),
            budget_bytes,
            remote: None,
            #[cfg(test)]
            parses: AtomicUsize::new(0),
        }
    }

    /// The cached footer of `object`. DataFusion keys the cache by the path
    /// inside the store, without the bucket, and validates an entry by size
    /// and modification time; an entry whose e_tag or version differs from
    /// the object's is also refused.
    fn remote_footer(&self, object: &ObjectMeta) -> Option<Arc<ParquetMetaData>> {
        let entry = self.remote.as_ref()?.get(&object.location)?;
        let same_identity = |cached: &Option<String>, current: &Option<String>| match (cached, current) {
            (Some(cached), Some(current)) => cached == current,
            _ => true,
        };
        if !entry.is_valid_for(object)
            || !same_identity(&entry.meta.e_tag, &object.e_tag)
            || !same_identity(&entry.meta.version, &object.version)
        {
            return None;
        }
        let footer = entry.file_metadata.as_any().downcast_ref::<CachedParquetMetaData>()?;
        Some(Arc::clone(footer.parquet_metadata()))
    }

    fn footer(&self, path: &Path) -> ParquetResult<Arc<ParquetMetaData>> {
        let file_metadata = fs::metadata(path)?;
        let key =
            FooterKey { path: path.to_path_buf(), size: file_metadata.len(), modified: file_metadata.modified()? };
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(footer) = state.footers.get(&key) {
            return Ok(Arc::clone(footer));
        }
        let footer = Arc::new(parse_footer(path)?);
        #[cfg(test)]
        self.parses.fetch_add(1, Ordering::Relaxed);
        let footer_bytes = footer.memory_size();
        if state.retained_bytes.saturating_add(footer_bytes) <= self.budget_bytes {
            state.retained_bytes += footer_bytes;
            state.footers.insert(key, Arc::clone(&footer));
        }
        Ok(footer)
    }

    #[cfg(test)]
    fn parses(&self) -> usize {
        self.parses.load(Ordering::Relaxed)
    }
}

fn parse_footer(path: &Path) -> ParquetResult<ParquetMetaData> {
    #[cfg(test)]
    THREAD_PARSES.set(THREAD_PARSES.get() + 1);
    ParquetMetaDataReader::new().parse_and_finish(&File::open(path)?)
}

/// Shares one [`FooterCache`] across every [`parquet_footer`] call on this
/// thread until the outermost scope drops. Nested scopes join the active cache.
///
/// The guard owns thread-local state, so it must drop on the thread that
/// created it; the raw-pointer marker keeps it `!Send`.
pub(crate) struct FooterScope {
    installed: bool,
    _same_thread: PhantomData<*const ()>,
}

impl FooterScope {
    pub(crate) fn enter() -> Self {
        Self::install_with(|| Arc::new(FooterCache::for_scope(None)))
    }

    /// Installs `cache` unless a scope is already active on this thread, so a
    /// caller that outlives one scope can share its footers with a later one.
    pub(crate) fn join(cache: Arc<FooterCache>) -> Self {
        Self::install_with(|| cache)
    }

    fn install_with(cache: impl FnOnce() -> Arc<FooterCache>) -> Self {
        let installed = ACTIVE_CACHE.with_borrow_mut(|active| {
            let installed = active.is_none();
            if installed {
                *active = Some(cache());
            }
            installed
        });
        Self { installed, _same_thread: PhantomData }
    }

    /// Number of footers parsed by the cache this scope shares.
    #[cfg(test)]
    pub(crate) fn parses(&self) -> usize {
        ACTIVE_CACHE.with_borrow(|active| active.as_ref().map_or(0, |cache| cache.parses()))
    }
}

impl Drop for FooterScope {
    fn drop(&mut self) {
        if self.installed {
            ACTIVE_CACHE.with_borrow_mut(Option::take);
        }
    }
}

/// Returns the footer of the local Parquet file at `path`, parsed at most once
/// per active [`FooterScope`].
pub(crate) fn parquet_footer(path: &Path) -> ParquetResult<Arc<ParquetMetaData>> {
    match ACTIVE_CACHE.with_borrow(Clone::clone) {
        Some(cache) => cache.footer(path),
        None => parse_footer(path).map(Arc::new),
    }
}

/// Why planning has no footer for a scan object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FooterUnavailable {
    /// The local file could not be opened or parsed.
    Local,
    /// The planning session's file metadata cache holds no footer for this
    /// version of the remote object, or the active scope has no such cache.
    RemoteNotCached,
}

/// Returns the footer of the Parquet object `object` in the store at
/// `object_store_url`: a `file://` object is parsed from the local file, any
/// other object is read from the active scope's remote cache.
pub(crate) fn object_parquet_footer(
    object_store_url: &ObjectStoreUrl,
    object: &ObjectMeta,
) -> Result<Arc<ParquetMetaData>, FooterUnavailable> {
    if object_store_url.as_str().starts_with("file://") {
        let path = local_object_path_to_filesystem(object.location.as_ref());
        return parquet_footer(path.as_path()).map_err(|_error| FooterUnavailable::Local);
    }
    ACTIVE_CACHE
        .with_borrow(Clone::clone)
        .and_then(|cache| cache.remote_footer(object))
        .ok_or(FooterUnavailable::RemoteNotCached)
}

#[cfg(test)]
mod tests {
    use arrow::{
        array::Int64Array,
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    };
    use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};

    use super::*;

    fn write_parquet(path: &Path, rows: i64) {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
        let batch =
            RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int64Array::from_iter_values(0..rows))]).unwrap();
        let props = WriterProperties::builder().set_max_row_group_row_count(Some(2)).build();
        let mut writer = ArrowWriter::try_new(File::create(path).unwrap(), schema, Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    #[test]
    fn rewritten_file_is_parsed_again() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("input.parquet");
        write_parquet(&path, 4);
        let cache = FooterCache::new(usize::MAX);

        assert_eq!(cache.footer(&path).unwrap().num_row_groups(), 2);
        write_parquet(&path, 8);

        assert_eq!(cache.footer(&path).unwrap().num_row_groups(), 4);
        assert_eq!(cache.parses(), 2);
    }

    #[test]
    fn footers_past_the_byte_budget_are_parsed_on_each_request() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("input.parquet");
        write_parquet(&path, 4);
        let retained = FooterCache::new(usize::MAX);
        let unretained = FooterCache::new(0);

        for _ in 0..2 {
            retained.footer(&path).unwrap();
            unretained.footer(&path).unwrap();
        }

        assert_eq!(retained.parses(), 1);
        assert_eq!(unretained.parses(), 2);
    }

    #[tokio::test]
    async fn remote_footers_come_from_the_session_cache_for_the_same_object_version() {
        use datafusion::{
            datasource::physical_plan::parquet::metadata::DFParquetMetadata,
            execution::runtime_env::RuntimeEnv,
            object_store::{ObjectStoreExt, memory::InMemory, path::Path as ObjectStorePath},
        };

        let mut bytes = Vec::new();
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
        let batch =
            RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int64Array::from_iter_values(0..4))]).unwrap();
        let props = WriterProperties::builder().set_max_row_group_row_count(Some(2)).build();
        let mut writer = ArrowWriter::try_new(&mut bytes, schema, Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        let store = InMemory::new();
        let location = ObjectStorePath::from("table/part-0.parquet");
        store.put(&location, bytes.into()).await.unwrap();
        let object = store.head(&location).await.unwrap();
        let session_cache = RuntimeEnv::default().cache_manager.get_file_metadata_cache();
        DFParquetMetadata::new(&store, &object)
            .with_file_metadata_cache(Some(Arc::clone(&session_cache)))
            .fetch_metadata()
            .await
            .unwrap();
        let s3 = ObjectStoreUrl::parse("s3://bucket").unwrap();
        let rewritten = ObjectMeta { e_tag: Some("rewritten".to_string()), ..object.clone() };

        {
            let _scope = FooterScope::enter();
            assert_eq!(object_parquet_footer(&s3, &object).unwrap_err(), FooterUnavailable::RemoteNotCached);
        }
        let _scope = FooterScope::join(Arc::new(FooterCache::for_scope(Some(session_cache))));
        assert_eq!(object_parquet_footer(&s3, &object).unwrap().num_row_groups(), 2);
        assert_eq!(
            object_parquet_footer(&s3, &rewritten).unwrap_err(),
            FooterUnavailable::RemoteNotCached,
            "an entry cached for another e_tag of the object must not serve its footer"
        );
    }
}

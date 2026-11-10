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
//! A scope entered with the planning session's DataFusion file metadata cache
//! looks up every footer there first. DataFusion fills that cache while it
//! infers a table's schema and statistics, and a local footer parsed here is
//! put back, so later queries of the session reuse it. Physical optimizer
//! rules run synchronously on the caller's Tokio worker, so planning never
//! reads a remote footer from the store; a remote object missing from that
//! cache has no footer here.

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
    execution::{
        cache::cache_manager::{CachedFileMetadataEntry, FileMetadataCache},
        object_store::ObjectStoreUrl,
    },
    object_store::{ObjectMeta, path::Path as ObjectStorePath},
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
    /// The planning session's DataFusion file metadata cache. It is the only
    /// source of remote footers and keeps parsed local footers across scopes.
    session: Option<Arc<FileMetadataCache>>,
    #[cfg(test)]
    parses: AtomicUsize,
}

impl std::fmt::Debug for FooterCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FooterCache")
            .field("budget_bytes", &self.budget_bytes)
            .field("session", &self.session.is_some())
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
    /// A cache with the budget every planning scope retains. `session` is the
    /// planning session's file metadata cache; without it, remote objects
    /// have no footer and local footers last only as long as the scope.
    pub(crate) fn for_scope(session: Option<Arc<FileMetadataCache>>) -> Self {
        Self { session, ..Self::new(FOOTER_CACHE_BUDGET_BYTES) }
    }

    fn new(budget_bytes: usize) -> Self {
        Self {
            state: Mutex::new(CacheState::default()),
            budget_bytes,
            session: None,
            #[cfg(test)]
            parses: AtomicUsize::new(0),
        }
    }

    /// The session cache's footer of `object`. DataFusion keys the cache by
    /// the path inside the store, without the bucket, and validates an entry
    /// by size and modification time; an entry whose e_tag or version differs
    /// from the object's is also refused.
    fn session_footer(&self, object: &ObjectMeta) -> Option<Arc<ParquetMetaData>> {
        let entry = self.session.as_ref()?.get(&object.location)?;
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

    /// The footer of the local file at `path`, which the store names
    /// `location`. A cached footer is current only for the size and
    /// modification time the file has on disk now, not the ones the scan
    /// listed, because execution reads the file as it is now.
    fn footer(&self, location: &ObjectStorePath, path: &Path) -> ParquetResult<Arc<ParquetMetaData>> {
        let file_metadata = fs::metadata(path)?;
        let modified = file_metadata.modified()?;
        let on_disk = ObjectMeta {
            location: location.clone(),
            last_modified: modified.into(),
            size: file_metadata.len(),
            e_tag: None,
            version: None,
        };
        if let Some(footer) = self.session_footer(&on_disk) {
            return Ok(footer);
        }
        let key = FooterKey { path: path.to_path_buf(), size: file_metadata.len(), modified };
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
        // DataFusion rejects an entry larger than its cache limit; the scope
        // map above still shares such a footer within this scope.
        if let Some(session) = &self.session {
            session.put(
                location,
                CachedFileMetadataEntry::new(on_disk, Arc::new(CachedParquetMetaData::new(Arc::clone(&footer)))),
            );
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
/// `object_store_url`. Both kinds of object are looked up in the active
/// scope's session cache first. A `file://` object missing there is parsed
/// from the local file, at most once per active [`FooterScope`]; any other
/// object has no footer.
pub(crate) fn object_parquet_footer(
    object_store_url: &ObjectStoreUrl,
    object: &ObjectMeta,
) -> Result<Arc<ParquetMetaData>, FooterUnavailable> {
    let cache = ACTIVE_CACHE.with_borrow(Clone::clone);
    if object_store_url.as_str().starts_with("file://") {
        let path = local_object_path_to_filesystem(object.location.as_ref());
        let footer = match cache {
            Some(cache) => cache.footer(&object.location, &path),
            None => parse_footer(&path).map(Arc::new),
        };
        return footer.map_err(|_error| FooterUnavailable::Local);
    }
    cache.and_then(|cache| cache.session_footer(object)).ok_or(FooterUnavailable::RemoteNotCached)
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

    fn location(path: &Path) -> ObjectStorePath {
        ObjectStorePath::from(path.to_str().unwrap())
    }

    #[test]
    fn rewritten_file_is_parsed_again() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("input.parquet");
        write_parquet(&path, 4);
        let cache = FooterCache::new(usize::MAX);

        assert_eq!(cache.footer(&location(&path), &path).unwrap().num_row_groups(), 2);
        write_parquet(&path, 8);

        assert_eq!(cache.footer(&location(&path), &path).unwrap().num_row_groups(), 4);
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
            retained.footer(&location(&path), &path).unwrap();
            unretained.footer(&location(&path), &path).unwrap();
        }

        assert_eq!(retained.parses(), 1);
        assert_eq!(unretained.parses(), 2);
    }

    #[tokio::test]
    async fn local_footers_outlive_the_scope_in_the_session_cache_until_the_file_changes() {
        use datafusion::{
            execution::runtime_env::RuntimeEnv,
            object_store::{ObjectStoreExt, local::LocalFileSystem},
        };

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("input.parquet");
        write_parquet(&path, 4);
        let store = LocalFileSystem::new();
        let local = ObjectStoreUrl::local_filesystem();
        let session_cache = RuntimeEnv::default().cache_manager.get_file_metadata_cache();
        let row_groups_in_new_scope = |object: &ObjectMeta| {
            let scope = FooterScope::join(Arc::new(FooterCache::for_scope(Some(Arc::clone(&session_cache)))));
            (object_parquet_footer(&local, object).unwrap().num_row_groups(), scope.parses())
        };

        let object = store.head(&location(&path)).await.unwrap();
        assert_eq!(row_groups_in_new_scope(&object), (2, 1));
        assert_eq!(row_groups_in_new_scope(&object), (2, 0), "a later scope of the session must not parse again");

        write_parquet(&path, 8);
        let rewritten = store.head(&location(&path)).await.unwrap();
        assert_eq!(row_groups_in_new_scope(&rewritten), (4, 1), "a rewritten file must be parsed again");
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

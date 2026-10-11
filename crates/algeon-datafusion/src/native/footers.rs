//! Parquet footers shared by the source-lowering steps of one planning run.
//!
//! One lowering reads the footer of a local file from several places (schema
//! mapping, byte-range coalescing, row-group pruning, byte metadata, source
//! diagnostics), and one optimizer run lowers the same plan more than once
//! (whole plan, mixed leaf, GPU relation inputs, graph residency). A
//! [`FooterScope`] entered on the planning thread lets every read in that
//! window share one parse per file. Reads outside a scope parse independently.

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

use parquet::{
    errors::Result as ParquetResult,
    file::metadata::{ParquetMetaData, ParquetMetaDataReader},
};

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
#[derive(Debug)]
pub(crate) struct FooterCache {
    state: Mutex<CacheState>,
    budget_bytes: usize,
    #[cfg(test)]
    parses: AtomicUsize,
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
    /// A cache with the budget every planning scope retains.
    pub(crate) fn for_scope() -> Self {
        Self::new(FOOTER_CACHE_BUDGET_BYTES)
    }

    fn new(budget_bytes: usize) -> Self {
        Self {
            state: Mutex::new(CacheState::default()),
            budget_bytes,
            #[cfg(test)]
            parses: AtomicUsize::new(0),
        }
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
        Self::install_with(|| Arc::new(FooterCache::for_scope()))
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
}

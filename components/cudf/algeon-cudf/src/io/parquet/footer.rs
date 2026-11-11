//! Parsed Parquet footers that several reads of one file share.

#[cfg(any(test, feature = "test-hooks"))]
use std::sync::atomic::{AtomicU64, Ordering};
use std::{
    fmt, fs,
    path::{Path, PathBuf},
    ptr::{self, NonNull},
    sync::{Arc, Mutex, OnceLock, PoisonError, mpsc},
    thread,
    time::SystemTime,
};

use super::super::{map_path_io_error, path_to_cstring};
use crate::{
    core::error::{self, Result},
    hash::FastHashMap,
};

/// The cuDF reader's parsed footer of one local Parquet file, page indexes
/// included.
pub(crate) struct ParquetFooter {
    handle: NonNull<cudf_sys::cudf_io_parquet_footer_handle_state>,
}

impl fmt::Debug for ParquetFooter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ParquetFooter").field("handle", &self.handle).finish()
    }
}

// SAFETY: The handle owns an immutable native footer. Reads only copy from
// it, so concurrent reads on several threads may share one footer.
unsafe impl Send for ParquetFooter {}
unsafe impl Sync for ParquetFooter {}

impl ParquetFooter {
    fn read(path: &Path, operation: &'static str) -> Result<Self> {
        let filepath = path_to_cstring(path, operation)?;
        let mut handle = ptr::null_mut();
        // SAFETY: `filepath` is a valid C string for the call and `handle` is
        // a valid output slot.
        let status = unsafe { cudf_sys::cudf_io_read_parquet_footer(filepath.as_ptr(), &mut handle) };
        error::check_status_in_with_details(status, operation, [("path", path.display().to_string())])?;
        Ok(Self { handle: error::require_non_null(handle, operation, "Parquet footer")? })
    }

    /// Borrowed native handle for one read call; `self` must outlive the call.
    pub(crate) fn raw_handle(&self) -> cudf_sys::cudf_io_parquet_footer_handle {
        self.handle.as_ptr()
    }
}

impl Drop for ParquetFooter {
    fn drop(&mut self) {
        release_on_footer_thread(FooterRelease(self.handle));
    }
}

/// A footer handle whose owner has dropped it.
struct FooterRelease(NonNull<cudf_sys::cudf_io_parquet_footer_handle_state>);

// SAFETY: The handle owns host memory only, which any thread may free.
unsafe impl Send for FooterRelease {}

impl FooterRelease {
    fn destroy(self) {
        // The footer is host memory, so a failed release reports nothing about
        // device health; the destroy ABI catches every native exception.
        // SAFETY: The handle came from `cudf_io_read_parquet_footer`, and its
        // only owner gave it up.
        let _ = unsafe { cudf_sys::cudf_io_parquet_footer_destroy(self.0.as_ptr()) };
    }
}

/// Frees `footer` on the process's footer release thread.
///
/// Freeing the footer of a file with thousands of row groups takes about as
/// long as parsing it, and the last owner usually drops it while its query
/// attempt ends. libcudf frees the footers it parses itself on its host worker
/// pool for the same reason. The thread starts on the first release; a
/// footer is freed in place when the thread cannot start.
fn release_on_footer_thread(footer: FooterRelease) {
    static RELEASE: OnceLock<Option<mpsc::Sender<FooterRelease>>> = OnceLock::new();
    let sender = RELEASE.get_or_init(|| {
        let (sender, receiver) = mpsc::channel::<FooterRelease>();
        thread::Builder::new()
            .name("algeon-parquet-footer-release".to_owned())
            .spawn(move || receiver.into_iter().for_each(FooterRelease::destroy))
            .ok()
            .map(|_| sender)
    });
    match sender {
        Some(sender) => {
            if let Err(mpsc::SendError(footer)) = sender.send(footer) {
                footer.destroy();
            }
        },
        None => footer.destroy(),
    }
}

/// Parsed footers of local Parquet files, for an owner that reads several
/// row-group ranges of the same files.
///
/// Without a cache each read of a file parses that file's whole footer again.
/// A footer is current for the size and modification time the file has on
/// disk when a read looks it up, so a file replaced since the last lookup is
/// parsed again. A rewrite that keeps both the size and the modification time
/// is not detected.
///
/// Retained footers are host memory that no device grant accounts for. They
/// are released when the cache drops, so the owner sets their lifetime.
#[derive(Debug, Default)]
pub struct ParquetFooterCache {
    footers: Mutex<FastHashMap<FooterKey, Arc<FooterSlot>>>,
    #[cfg(any(test, feature = "test-hooks"))]
    parses: AtomicU64,
}

/// One file's footer, filled by the first read that needs it. Concurrent
/// reads of the same file wait on the slot instead of parsing it twice.
type FooterSlot = Mutex<Option<Arc<ParquetFooter>>>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FooterKey {
    path: PathBuf,
    size: u64,
    modified: Option<SystemTime>,
}

impl ParquetFooterCache {
    /// An empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// The footer of the local file at `path` as it is on disk now.
    ///
    /// A footer another read already parsed for the same file state is
    /// shared. A newly parsed footer is kept for later reads only when
    /// `retain` is true; a read that covers the whole file passes false, since
    /// no other chunk of that file will ask for it.
    pub(crate) fn footer(&self, path: &Path, retain: bool, operation: &'static str) -> Result<Arc<ParquetFooter>> {
        let metadata = fs::metadata(path)
            .map_err(|source| map_path_io_error(source, path, operation, "failed to access input path"))?;
        let key = FooterKey { path: path.to_path_buf(), size: metadata.len(), modified: metadata.modified().ok() };
        let slot = {
            let mut footers = self.footers.lock().unwrap_or_else(PoisonError::into_inner);
            match footers.get(&key) {
                Some(slot) => Some(Arc::clone(slot)),
                None if retain => Some(Arc::clone(footers.entry(key).or_default())),
                None => None,
            }
        };
        let Some(slot) = slot else {
            return self.parse(path, operation).map(Arc::new);
        };
        let mut footer = slot.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(footer) = footer.as_ref() {
            return Ok(Arc::clone(footer));
        }
        let parsed = Arc::new(self.parse(path, operation)?);
        *footer = Some(Arc::clone(&parsed));
        Ok(parsed)
    }

    fn parse(&self, path: &Path, operation: &'static str) -> Result<ParquetFooter> {
        let footer = ParquetFooter::read(path, operation)?;
        #[cfg(any(test, feature = "test-hooks"))]
        self.parses.fetch_add(1, Ordering::Relaxed);
        Ok(footer)
    }

    /// How many footers this cache has parsed, retained or not.
    #[cfg(any(test, feature = "test-hooks"))]
    #[doc(hidden)]
    pub fn parsed_footer_count(&self) -> u64 {
        self.parses.load(Ordering::Relaxed)
    }
}

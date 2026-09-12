//! Parquet footers supplied to planning by whoever read them.
//!
//! Planning consumes footers; it does not open the files they came from. The
//! caller; source preparation, which already loads and caches these footers;
//! passes what it has, so a footer is read once per source rather than once per
//! decision that happens to need it.

use parquet::file::metadata::ParquetMetaData;

/// Footers the caller already loaded, one slot per file of a
/// [`LocalParquetSource`](super::LocalParquetSource), in `files` order.
///
/// A `None` slot means the caller has no footer for that file, and every
/// decision that needs one skips rather than guessing. That keeps the
/// conservative outcome the previous in-planning read produced for a source it
/// could not open: the optimization does not apply, and the plan stays correct.
/// Whether an unreadable file is an error at all is source preparation's call,
/// not a pruning decision's.
#[derive(Debug, Clone, Copy, Default)]
pub struct SourceFooterFacts<'a> {
    per_file: &'a [Option<&'a ParquetMetaData>],
}

impl<'a> SourceFooterFacts<'a> {
    /// Supplies one footer slot per source file, in `files` order.
    pub const fn new(per_file: &'a [Option<&'a ParquetMetaData>]) -> Self {
        Self { per_file }
    }

    /// Supplies no footers, so every footer-dependent decision is skipped.
    pub const fn none() -> Self {
        Self { per_file: &[] }
    }

    /// The footer for `file_index`, when the caller supplied one.
    pub fn footer(&self, file_index: usize) -> Option<&'a ParquetMetaData> {
        self.per_file.get(file_index).copied().flatten()
    }
}

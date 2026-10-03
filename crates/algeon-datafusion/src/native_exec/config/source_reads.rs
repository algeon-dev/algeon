//! Remote Parquet read settings, fixed when the execution config is built.

use std::{
    num::NonZeroU64,
    path::{Path, PathBuf},
};

/// When a local byte-cache miss for an `s3://` Parquet object is filled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum LakehouseByteCacheFill {
    /// A miss reads the object remotely and leaves the cache unchanged.
    #[default]
    Disabled,
    /// A miss reads the object remotely and fills the cache in the background.
    ReadThrough,
    /// A miss fills the cache first; the read then opens the local file.
    FillBeforeRead,
}

/// Local byte cache for `s3://` Parquet objects.
///
/// Only an object with an ETag or version is served from or filled into the
/// cache; its entry is keyed by that identity, the bucket and endpoint, and
/// the object location.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LakehouseByteCache {
    root: PathBuf,
    serves_hits: bool,
    fill: LakehouseByteCacheFill,
    max_retained_bytes: Option<NonZeroU64>,
}

impl LakehouseByteCache {
    /// A cache under `root` that serves complete entries and never fills.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into(), serves_hits: true, fill: LakehouseByteCacheFill::Disabled, max_retained_bytes: None }
    }

    /// Selects when a miss is filled.
    pub fn with_fill(mut self, fill: LakehouseByteCacheFill) -> Self {
        self.fill = fill;
        self
    }

    /// Reads every object remotely even when a complete entry exists, for
    /// benchmark isolation. Misses are still filled as [`Self::with_fill`]
    /// selects.
    pub fn ignoring_hits(mut self) -> Self {
        self.serves_hits = false;
        self
    }

    /// Best-effort byte target the cache root is trimmed toward.
    pub fn with_max_retained_bytes(mut self, bytes: NonZeroU64) -> Self {
        self.max_retained_bytes = Some(bytes);
        self
    }

    pub(crate) fn root(&self) -> &Path {
        &self.root
    }

    pub(crate) fn serves_hits(&self) -> bool {
        self.serves_hits
    }

    pub(crate) fn fill(&self) -> LakehouseByteCacheFill {
        self.fill
    }

    pub(crate) fn max_retained_bytes(&self) -> Option<NonZeroU64> {
        self.max_retained_bytes
    }
}

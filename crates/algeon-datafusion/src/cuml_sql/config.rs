use std::path::PathBuf;

/// Session configuration for cuML SQL.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CumlSqlConfig {
    /// Directory `cuml_forest_predict` resolves its `model` argument under.
    /// Without it, `cuml_forest_predict` reports `model_root_unconfigured`.
    pub model_store: Option<CumlModelStoreConfig>,
}

/// Models root and per-file size bound for `cuml_forest_predict`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CumlModelStoreConfig {
    /// Existing directory. A `model` argument must resolve to a regular file
    /// beneath it; symlinks are followed only when they are relative and stay
    /// beneath it. The directory must not be writable by untrusted users.
    pub root: PathBuf,
    /// Model files larger than this are rejected with `model_too_large`
    /// before parsing.
    pub max_model_bytes: u64,
}

impl CumlModelStoreConfig {
    pub const DEFAULT_MAX_MODEL_BYTES: u64 = 256 * 1024 * 1024;

    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into(), max_model_bytes: Self::DEFAULT_MAX_MODEL_BYTES }
    }
}

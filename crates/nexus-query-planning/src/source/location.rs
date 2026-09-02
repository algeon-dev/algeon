//! Source location and read-capability contracts.

use std::{
    fmt,
    path::{Path, PathBuf},
};

use super::diagnostics::{
    object_store_diagnostic_id, redacted_debug_value, redacted_stable_id_diagnostic, stable_fnv1a64,
};

/// Explicit location/read-capability model for native Parquet sources.
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum ParquetSourceLocation {
    /// Local filesystem path readable by cuDF local Parquet readers.
    LocalPath(PathBuf),
    /// Object-store URI plus parsed scheme.
    ObjectStoreUri {
        /// Lowercase object-store scheme such as `s3`.
        scheme: String,
        /// Original object-store URI.
        uri: String,
    },
}

impl fmt::Debug for ParquetSourceLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LocalPath(path) => f
                .debug_tuple("LocalPath")
                .field(&redacted_debug_value("path", path.display().to_string().as_str()))
                .finish(),
            Self::ObjectStoreUri { scheme, uri } => f
                .debug_struct("ObjectStoreUri")
                .field("scheme", scheme)
                .field("uri", &redacted_debug_value("uri", uri))
                .finish(),
        }
    }
}

impl ParquetSourceLocation {
    /// Build a local filesystem source location.
    pub fn local_path(path: impl Into<PathBuf>) -> Self {
        Self::LocalPath(path.into())
    }

    /// Build a supported object-store URI location.
    ///
    /// Currently only `s3://...` is accepted. Errors are stable reason strings
    /// such as `missing_uri_scheme` or `unsupported_uri_scheme:<scheme>`.
    pub fn object_store_uri(uri: impl Into<String>) -> Result<Self, String> {
        let uri = uri.into();
        let Some((scheme, _)) = uri.split_once("://") else {
            return Err("missing_uri_scheme".to_string());
        };
        if scheme.is_empty() {
            return Err("empty_uri_scheme".to_string());
        }
        let scheme = scheme.to_ascii_lowercase();
        if scheme != "s3" {
            return Err(format!("unsupported_uri_scheme:{scheme}"));
        }
        Ok(Self::ObjectStoreUri { scheme, uri })
    }

    /// Path or URI string passed to lower-level readers.
    pub fn reader_path(&self) -> String {
        match self {
            Self::LocalPath(path) => path.display().to_string(),
            Self::ObjectStoreUri { uri, .. } => uri.clone(),
        }
    }

    /// Stable read-capability diagnostic string.
    pub fn read_capability(&self) -> String {
        match self {
            Self::LocalPath(_) => "local_file_system".to_string(),
            Self::ObjectStoreUri { scheme, .. } => format!("object_store:{scheme}"),
        }
    }

    /// Whether this location is a local filesystem path.
    pub fn is_local_path(&self) -> bool {
        matches!(self, Self::LocalPath(_))
    }

    /// Borrow the local path when this location is local.
    pub fn as_local_path(&self) -> Option<&Path> {
        match self {
            Self::LocalPath(path) => Some(path.as_path()),
            Self::ObjectStoreUri { .. } => None,
        }
    }

    /// Display wrapper for path/URI diagnostics.
    pub fn display(&self) -> impl fmt::Display + '_ {
        self
    }

    /// Redacted diagnostic location. Local paths are preserved; object-store
    /// URIs are represented by scheme plus stable hash only.
    pub fn diagnostic_location(&self) -> String {
        match self {
            Self::LocalPath(path) => path.display().to_string(),
            Self::ObjectStoreUri { scheme, uri } => object_store_diagnostic_id(scheme, uri),
        }
    }

    #[doc(hidden)]
    pub fn diagnostic_stable_id(&self, stable_id: &str) -> String {
        match self {
            Self::LocalPath(_) => redacted_stable_id_diagnostic(stable_id),
            Self::ObjectStoreUri { scheme, .. } => {
                format!("object_store:{scheme}:stable_id_fnv64:{:016x}", stable_fnv1a64(stable_id.as_bytes()))
            },
        }
    }
}

impl fmt::Display for ParquetSourceLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LocalPath(path) => write!(f, "{}", path.display()),
            Self::ObjectStoreUri { uri, .. } => f.write_str(uri),
        }
    }
}

pub(super) fn parquet_source_location_for_original(location: &str) -> Result<ParquetSourceLocation, String> {
    if let Some(path) = location.strip_prefix("file://localhost/") {
        return Ok(ParquetSourceLocation::local_path(format!("/{path}")));
    }
    if let Some(path) = location.strip_prefix("file://") {
        return Ok(ParquetSourceLocation::local_path(path));
    }
    if let Some(path) = location.strip_prefix("file:") {
        return Ok(ParquetSourceLocation::local_path(path));
    }
    if location.contains("://") {
        return ParquetSourceLocation::object_store_uri(location);
    }
    Ok(ParquetSourceLocation::local_path(location))
}

pub(super) fn object_store_reader_path(uri: &str) -> String {
    uri.to_string()
}

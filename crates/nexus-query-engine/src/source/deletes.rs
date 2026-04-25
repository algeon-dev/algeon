//! Iceberg row-delete file source contracts.

use std::{fmt, path::PathBuf};

use super::{
    diagnostics::{redacted_debug_option, redacted_debug_value},
    location::{ParquetSourceLocation, parquet_source_location_for_original},
    parquet::LocalParquetFile,
    read_policy::ResolvedParquetSource,
};

/// Iceberg position-delete Parquet file associated with one data file.
///
/// The delete file is read as a normal native Parquet source and filtered by
/// Iceberg's `file_path` column before anti-joining data rows by file-global
/// `pos`.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ParquetPositionDeleteFile {
    /// Stable delete-file id for diagnostics.
    pub id: String,
    /// Explicit source location for the delete Parquet file.
    pub location: ParquetSourceLocation,
    /// Path-shaped reader string retained for diagnostics and cuDF reader APIs.
    pub path: PathBuf,
    /// Resolved reader strategy and decode contract for this delete file.
    pub resolved_source: ResolvedParquetSource,
    /// Stable identifier for diagnostics and executor bookkeeping.
    pub stable_id: String,
    /// Known delete-file size in bytes, when available.
    pub file_size_bytes: Option<u64>,
    /// Known delete row count, when available.
    pub row_count: Option<u64>,
    /// Referenced data-file path from Iceberg position-delete metadata, when present.
    pub referenced_data_file: Option<String>,
    /// Inclusive lower bound for the Iceberg `pos` column, when manifest
    /// metrics prove it.
    pub position_lower_bound: Option<u64>,
    /// Inclusive upper bound for the Iceberg `pos` column, when manifest
    /// metrics prove it.
    pub position_upper_bound: Option<u64>,
}

impl fmt::Debug for ParquetPositionDeleteFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ParquetPositionDeleteFile")
            .field("id", &self.id)
            .field("location", &self.location)
            .field("path", &redacted_debug_value("path", self.path.display().to_string().as_str()))
            .field("resolved_source", &self.resolved_source)
            .field("stable_id", &self.location.diagnostic_stable_id(self.stable_id.as_str()))
            .field("file_size_bytes", &self.file_size_bytes)
            .field("row_count", &self.row_count)
            .field(
                "referenced_data_file",
                &redacted_debug_option("referenced_data_file", self.referenced_data_file.as_deref()),
            )
            .field("position_lower_bound", &self.position_lower_bound)
            .field("position_upper_bound", &self.position_upper_bound)
            .finish()
    }
}

impl ParquetPositionDeleteFile {
    /// Creates a delete-file descriptor from an already resolved reader contract.
    pub fn from_resolved_parquet_source(
        id: impl Into<String>,
        resolved_source: ResolvedParquetSource,
    ) -> Result<Self, String> {
        let id = id.into();
        let location = parquet_source_location_for_original(&resolved_source.original_location)?;
        let stable_id = format!("position-delete:{id}:{}", location.diagnostic_location());
        Ok(Self {
            id,
            location,
            path: PathBuf::from(resolved_source.reader_location()),
            resolved_source,
            stable_id,
            file_size_bytes: None,
            row_count: None,
            referenced_data_file: None,
            position_lower_bound: None,
            position_upper_bound: None,
        })
    }

    /// Sets known delete-file size in bytes.
    pub fn with_file_size_bytes(mut self, file_size_bytes: u64) -> Self {
        self.file_size_bytes = Some(file_size_bytes);
        self
    }

    /// Sets known delete row count.
    pub fn with_row_count(mut self, row_count: u64) -> Self {
        self.row_count = Some(row_count);
        self
    }

    /// Sets the referenced data-file path for manifest-proven position-delete pruning.
    pub fn with_referenced_data_file(mut self, referenced_data_file: impl Into<String>) -> Self {
        self.referenced_data_file = Some(referenced_data_file.into());
        self
    }

    /// Sets inclusive Iceberg `pos` bounds from delete-file manifest metrics.
    pub fn with_position_bounds(mut self, lower_bound: Option<u64>, upper_bound: Option<u64>) -> Self {
        self.position_lower_bound = lower_bound;
        self.position_upper_bound = upper_bound;
        self
    }

    /// Builds a normal local Parquet file descriptor for reading the delete file.
    pub fn to_local_parquet_file(&self) -> LocalParquetFile {
        let mut file = LocalParquetFile::new(self.path.clone())
            .with_resolved_parquet_source(self.resolved_source.clone())
            .expect("validated position delete source remains representable")
            .with_stable_id(self.stable_id.clone());
        if let Some(file_size_bytes) = self.file_size_bytes {
            file = file.with_file_size_bytes(file_size_bytes);
        }
        if let Some(row_count) = self.row_count {
            file = file.with_row_group_rows(vec![row_count]);
        }
        file
    }
}

/// Iceberg equality-delete Parquet file associated with one data file.
///
/// The delete file is read as a normal native Parquet source and anti-joined
/// against the decoded data rows using Iceberg equality semantics: NULL keys
/// match NULL keys.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ParquetEqualityDeleteFile {
    /// Stable delete-file id for diagnostics.
    pub id: String,
    /// Explicit source location for the delete Parquet file.
    pub location: ParquetSourceLocation,
    /// Path-shaped reader string retained for diagnostics and cuDF reader APIs.
    pub path: PathBuf,
    /// Resolved reader strategy and decode contract for this delete file.
    pub resolved_source: ResolvedParquetSource,
    /// Stable identifier for diagnostics and executor bookkeeping.
    pub stable_id: String,
    /// Iceberg field ids used as equality keys.
    pub equality_ids: Vec<i32>,
    /// Column names to read from the data file and delete file.
    pub equality_columns: Vec<String>,
    /// Known delete-file size in bytes, when available.
    pub file_size_bytes: Option<u64>,
    /// Known delete row count, when available.
    pub row_count: Option<u64>,
}

impl fmt::Debug for ParquetEqualityDeleteFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ParquetEqualityDeleteFile")
            .field("id", &self.id)
            .field("location", &self.location)
            .field("path", &redacted_debug_value("path", self.path.display().to_string().as_str()))
            .field("resolved_source", &self.resolved_source)
            .field("stable_id", &self.location.diagnostic_stable_id(self.stable_id.as_str()))
            .field("equality_ids", &self.equality_ids)
            .field("equality_columns", &self.equality_columns)
            .field("file_size_bytes", &self.file_size_bytes)
            .field("row_count", &self.row_count)
            .finish()
    }
}

impl ParquetEqualityDeleteFile {
    /// Creates a delete-file descriptor from an already resolved reader contract.
    pub fn from_resolved_parquet_source(
        id: impl Into<String>,
        resolved_source: ResolvedParquetSource,
        equality_ids: Vec<i32>,
        equality_columns: Vec<String>,
    ) -> Result<Self, String> {
        if equality_ids.is_empty() || equality_ids.len() != equality_columns.len() {
            return Err("equality delete files require matching non-empty equality ids and columns".to_string());
        }
        let id = id.into();
        let location = parquet_source_location_for_original(&resolved_source.original_location)?;
        let stable_id = format!("equality-delete:{id}:{}", location.diagnostic_location());
        Ok(Self {
            id,
            location,
            path: PathBuf::from(resolved_source.reader_location()),
            resolved_source,
            stable_id,
            equality_ids,
            equality_columns,
            file_size_bytes: None,
            row_count: None,
        })
    }

    /// Sets known delete-file size in bytes.
    pub fn with_file_size_bytes(mut self, file_size_bytes: u64) -> Self {
        self.file_size_bytes = Some(file_size_bytes);
        self
    }

    /// Sets known delete row count.
    pub fn with_row_count(mut self, row_count: u64) -> Self {
        self.row_count = Some(row_count);
        self
    }

    /// Builds a normal local Parquet file descriptor for reading the delete file.
    pub fn to_local_parquet_file(&self) -> LocalParquetFile {
        let mut file = LocalParquetFile::new(self.path.clone())
            .with_resolved_parquet_source(self.resolved_source.clone())
            .expect("validated equality delete source remains representable")
            .with_stable_id(self.stable_id.clone());
        if let Some(file_size_bytes) = self.file_size_bytes {
            file = file.with_file_size_bytes(file_size_bytes);
        }
        if let Some(row_count) = self.row_count {
            file = file.with_row_group_rows(vec![row_count]);
        }
        file
    }
}

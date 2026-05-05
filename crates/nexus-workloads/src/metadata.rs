use std::path::PathBuf;

use serde::Serialize;

/// Adapter-neutral dataset metadata shared by workload consumers.
///
/// Field meanings and ordering mirror the stable report header shape; the
/// bench/report layer owns conversion into adapter report types.
/// This is the intentionally small, serde-stable projection used by external
/// evidence artifacts. Workload-specific discovery types retain their richer
/// table-level details and project here at the public reporting boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WorkloadDatasetMetadata {
    pub workload: &'static str,
    pub root: PathBuf,
    pub scale_factor: Option<String>,
    pub provenance: String,
    pub generation_command: String,
    pub table_count: usize,
    pub file_count: usize,
    pub total_rows: u64,
    pub total_bytes: u64,
    pub metadata_fingerprint: String,
    pub content_fingerprint: Option<String>,
    pub table_file_counts: String,
    pub table_row_counts: String,
    pub table_byte_sizes: String,
    pub table_row_group_counts: String,
}

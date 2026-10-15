use std::{hash::Hasher, path::PathBuf};

use fnv::FnvHasher;
use serde::Serialize;

pub(crate) struct DatasetFingerprint(FnvHasher);

impl DatasetFingerprint {
    pub(crate) fn new() -> Self {
        Self(FnvHasher::default())
    }

    pub(crate) fn update_field(&mut self, value: &str) {
        self.update_bytes(value.as_bytes());
        self.update_bytes(&[0]);
    }

    pub(crate) fn update_bytes(&mut self, bytes: &[u8]) {
        self.0.write(bytes);
    }

    pub(crate) fn finish(&self) -> u64 {
        self.0.finish()
    }
}

/// Adapter-neutral dataset metadata shared by workload consumers.
///
/// Field names and order follow the report header schema. Workload-specific
/// discovery structures retain per-table details and serialize into this
/// projection when producing external evidence artifacts.
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

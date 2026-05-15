use std::path::PathBuf;

/// Workload-agnostic dataset metadata for shared report TSV output.
///
/// Per-workload modules own their typed metadata types (e.g. `TpchDatasetMetadata`).
/// They project into `WorkloadDatasetMetadata` for embedding into shared report
/// surfaces (`NativeFallbackPressureReport`, `NativeDataFusionPlanningReport`,
/// `NativeDataFusionExecutionReport`). The per-table breakdown fields are
/// pre-formatted CSV strings rather than typed table lists so the shared report
/// surface does not have to know about workload-specific table schemas.
///
/// `scale_factor` is `Option<String>` because not every workload has the
/// TPC-style scale-factor concept (e.g. ClickBench).
#[derive(Debug, Clone, PartialEq, Eq)]
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

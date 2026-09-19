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
    /// Stable workload label such as `tpch`, `tpcds`, or `clickbench`.
    pub workload: &'static str,
    /// Fixture root used by the workload run.
    pub root: PathBuf,
    /// Optional scale/layout label; workloads without a TPC-style scale may use `None`.
    pub scale_factor: Option<String>,
    /// Stable provenance label for the fixture source.
    pub provenance: String,
    /// Free-form generation command or fixture description.
    pub generation_command: String,
    /// Number of logical tables in the fixture.
    pub table_count: usize,
    /// Total Parquet file count across the fixture.
    pub file_count: usize,
    /// Total row count from Parquet metadata.
    pub total_rows: u64,
    /// Total file bytes from filesystem metadata.
    pub total_bytes: u64,
    /// Stable metadata fingerprint produced by the workload-specific metadata type.
    pub metadata_fingerprint: String,
    /// Optional content fingerprint; missing values serialize as `not_computed`.
    pub content_fingerprint: Option<String>,
    /// CSV `table=count` summary for file counts.
    pub table_file_counts: String,
    /// CSV `table=count` summary for row counts.
    pub table_row_counts: String,
    /// CSV `table=bytes` summary for file bytes.
    pub table_byte_sizes: String,
    /// CSV `table=count` summary for row-group counts.
    pub table_row_group_counts: String,
}

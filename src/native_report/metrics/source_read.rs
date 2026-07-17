use std::collections::BTreeMap;

use datafusion::physical_plan::metrics::MetricsSet;

use super::labels::{bool_label, labels_by_name, optional_usize, optional_usize_label, string_label, usize_label};

const NATIVE_SOURCE_READ_RUNTIME_ROWS_METRIC_NAME: &str = "native_source_read_runtime_rows";

/// Stable nested metric row for one local-Parquet source read.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NativeSourceReadRuntimeMetricRow {
    /// Native source node id.
    pub source_node_id: String,
    /// Source read shape, for example `chunk` or `chunk_batch`.
    pub read_kind: String,
    /// Parquet read strategy used by the chunk or batch.
    pub read_strategy: String,
    /// Input kind passed to the Parquet reader.
    pub parquet_read_input_kind: String,
    /// Requested remote read mode, or `none` for non-remote reads.
    pub remote_read_mode: String,
    /// Parquet reader backend evidence.
    pub parquet_read_backend: String,
    /// Build-time remote backend status reported by cudf-rs/libcudf.
    pub remote_backend_build_status: String,
    /// Remote bytes read, when reported by the lower layer.
    pub remote_read_bytes: Option<usize>,
    /// Remote range count, when reported by the lower layer.
    pub remote_read_range_count: Option<usize>,
    /// Direct device-read support evidence reported by the lower layer.
    pub device_read_support: String,
    /// Row groups participating in this read after any caller row-group selection.
    pub parquet_pruning_input_row_groups: Option<usize>,
    /// Row groups surviving libcudf statistics pruning, when reported.
    pub parquet_row_groups_after_stats_filter: Option<usize>,
    /// Row groups surviving libcudf Bloom-filter pruning, when reported.
    pub parquet_row_groups_after_bloom_filter: Option<usize>,
    /// Stable status explaining Bloom-pruning evidence or known ineligibility.
    pub parquet_bloom_pruning_status: String,
    /// Batch index within the source stream, or chunk index for singleton reads.
    pub batch_index: usize,
    /// Number of source chunks represented by this read.
    pub chunk_count: usize,
    /// First source chunk index represented by this read.
    pub first_chunk_index: usize,
    /// Last source chunk index represented by this read.
    pub last_chunk_index: usize,
    /// First source file index represented by this read.
    pub first_file_index: usize,
    /// Last source file index represented by this read.
    pub last_file_index: usize,
    /// Number of distinct files represented by this read.
    pub file_count: usize,
    /// Selected row groups represented by this read.
    pub row_group_count: usize,
    /// Number of physical read columns per represented file.
    pub read_columns_per_file: usize,
    /// Sum of physical read-column selections across represented files.
    pub read_columns_total: usize,
    /// Sum of known file bytes for represented files, when available.
    pub known_file_bytes: usize,
    /// Best available source-read byte estimate for represented chunks.
    pub estimated_source_read_bytes: usize,
    /// Sum of known selected row-group bytes, when available.
    pub known_row_group_bytes: usize,
    /// Sum of known projected compressed bytes, when available.
    pub known_projected_compressed_bytes: usize,
    /// Sum of known projected uncompressed bytes, when available.
    pub known_projected_uncompressed_bytes: usize,
    /// Sum of known pure-filter compressed bytes, when available.
    pub known_pure_filter_compressed_bytes: usize,
    /// Sum of known pure-filter uncompressed bytes, when available.
    pub known_pure_filter_uncompressed_bytes: usize,
    /// Number of output rows materialized by the read after reader predicates.
    pub output_rows: usize,
    /// Whether this read carried a source reader predicate.
    pub reader_predicate: bool,
    /// Number of dynamic-filter reader predicate routes attached to this read.
    pub reader_dynamic_filter_count: usize,
    /// Stable diagnostic reason for the source chunk batching decision.
    pub source_batching_reason: String,
    /// Source chunk planning mode requested for this source read.
    pub source_chunk_planning_mode: String,
    /// Stable diagnostic reason for the source chunk row-group and byte target.
    pub source_chunk_planning_reason: String,
    /// Elapsed time spent in the source read, in nanoseconds.
    pub elapsed_ns: usize,
}

impl NativeSourceReadRuntimeMetricRow {
    /// Serializes this nested row as stable comma-separated key/value pairs.
    pub fn to_stable_kv_row(&self) -> String {
        format!(
            "source_node_id={},read_kind={},read_strategy={},batch_index={},chunk_count={},first_chunk_index={},last_chunk_index={},first_file_index={},last_file_index={},file_count={},row_group_count={},read_columns_per_file={},read_columns_total={},known_file_bytes={},estimated_source_read_bytes={},known_row_group_bytes={},known_projected_compressed_bytes={},known_projected_uncompressed_bytes={},known_pure_filter_compressed_bytes={},known_pure_filter_uncompressed_bytes={},output_rows={},reader_predicate={},reader_dynamic_filter_count={},source_batching_reason={},source_chunk_planning_mode={},source_chunk_planning_reason={},elapsed_ns={},parquet_read_input_kind={},remote_read_mode={},parquet_read_backend={},remote_backend_build_status={},remote_read_bytes={},remote_read_range_count={},device_read_support={},parquet_pruning_input_row_groups={},parquet_row_groups_after_stats_filter={},parquet_row_groups_after_bloom_filter={},parquet_bloom_pruning_status={}",
            self.source_node_id,
            self.read_kind,
            self.read_strategy,
            self.batch_index,
            self.chunk_count,
            self.first_chunk_index,
            self.last_chunk_index,
            self.first_file_index,
            self.last_file_index,
            self.file_count,
            self.row_group_count,
            self.read_columns_per_file,
            self.read_columns_total,
            self.known_file_bytes,
            self.estimated_source_read_bytes,
            self.known_row_group_bytes,
            self.known_projected_compressed_bytes,
            self.known_projected_uncompressed_bytes,
            self.known_pure_filter_compressed_bytes,
            self.known_pure_filter_uncompressed_bytes,
            self.output_rows,
            self.reader_predicate,
            self.reader_dynamic_filter_count,
            self.source_batching_reason,
            self.source_chunk_planning_mode,
            self.source_chunk_planning_reason,
            self.elapsed_ns,
            self.parquet_read_input_kind,
            self.remote_read_mode,
            self.parquet_read_backend,
            self.remote_backend_build_status,
            optional_usize(self.remote_read_bytes),
            optional_usize(self.remote_read_range_count),
            self.device_read_support,
            optional_usize(self.parquet_pruning_input_row_groups),
            optional_usize(self.parquet_row_groups_after_stats_filter),
            optional_usize(self.parquet_row_groups_after_bloom_filter),
            self.parquet_bloom_pruning_status,
        )
    }
}

pub(super) fn source_read_runtime_rows(metrics: &MetricsSet) -> Vec<NativeSourceReadRuntimeMetricRow> {
    let mut rows = metrics
        .iter()
        .filter(|metric| metric.value().name() == NATIVE_SOURCE_READ_RUNTIME_ROWS_METRIC_NAME)
        .filter(|metric| metric.labels().iter().any(|label| label.name() == "source_node_id"))
        .map(|metric| labels_to_source_read_runtime_row(labels_by_name(metric.labels())))
        .collect::<Vec<_>>();
    rows.sort_by_key(|row| row.to_stable_kv_row());
    rows
}

fn labels_to_source_read_runtime_row(labels: BTreeMap<&str, &str>) -> NativeSourceReadRuntimeMetricRow {
    NativeSourceReadRuntimeMetricRow {
        source_node_id: string_label(&labels, "source_node_id"),
        read_kind: string_label(&labels, "read_kind"),
        read_strategy: string_label(&labels, "read_strategy"),
        parquet_read_input_kind: optional_string_label_or_none(&labels, "parquet_read_input_kind"),
        remote_read_mode: optional_string_label_or_none(&labels, "remote_read_mode"),
        parquet_read_backend: optional_string_label_or_none(&labels, "parquet_read_backend"),
        remote_backend_build_status: optional_string_label_or_none(&labels, "remote_backend_build_status"),
        remote_read_bytes: optional_usize_label(&labels, "remote_read_bytes"),
        remote_read_range_count: optional_usize_label(&labels, "remote_read_range_count"),
        device_read_support: optional_string_label_or_none(&labels, "device_read_support"),
        parquet_pruning_input_row_groups: optional_usize_label(&labels, "parquet_pruning_input_row_groups"),
        parquet_row_groups_after_stats_filter: optional_usize_label(&labels, "parquet_row_groups_after_stats_filter"),
        parquet_row_groups_after_bloom_filter: optional_usize_label(&labels, "parquet_row_groups_after_bloom_filter"),
        parquet_bloom_pruning_status: optional_string_label_or_none(&labels, "parquet_bloom_pruning_status"),
        batch_index: usize_label(&labels, "batch_index"),
        chunk_count: usize_label(&labels, "chunk_count"),
        first_chunk_index: usize_label(&labels, "first_chunk_index"),
        last_chunk_index: usize_label(&labels, "last_chunk_index"),
        first_file_index: usize_label(&labels, "first_file_index"),
        last_file_index: usize_label(&labels, "last_file_index"),
        file_count: usize_label(&labels, "file_count"),
        row_group_count: usize_label(&labels, "row_group_count"),
        read_columns_per_file: usize_label(&labels, "read_columns_per_file"),
        read_columns_total: usize_label(&labels, "read_columns_total"),
        known_file_bytes: usize_label(&labels, "known_file_bytes"),
        estimated_source_read_bytes: usize_label(&labels, "estimated_source_read_bytes"),
        known_row_group_bytes: usize_label(&labels, "known_row_group_bytes"),
        known_projected_compressed_bytes: usize_label(&labels, "known_projected_compressed_bytes"),
        known_projected_uncompressed_bytes: usize_label(&labels, "known_projected_uncompressed_bytes"),
        known_pure_filter_compressed_bytes: usize_label(&labels, "known_pure_filter_compressed_bytes"),
        known_pure_filter_uncompressed_bytes: usize_label(&labels, "known_pure_filter_uncompressed_bytes"),
        output_rows: usize_label(&labels, "output_rows"),
        reader_predicate: bool_label(&labels, "reader_predicate"),
        reader_dynamic_filter_count: usize_label(&labels, "reader_dynamic_filter_count"),
        source_batching_reason: string_label(&labels, "source_batching_reason"),
        source_chunk_planning_mode: string_label(&labels, "source_chunk_planning_mode"),
        source_chunk_planning_reason: string_label(&labels, "source_chunk_planning_reason"),
        elapsed_ns: usize_label(&labels, "elapsed_ns"),
    }
}

fn optional_string_label_or_none(labels: &BTreeMap<&str, &str>, name: &str) -> String {
    labels.get(name).copied().unwrap_or("none").to_string()
}

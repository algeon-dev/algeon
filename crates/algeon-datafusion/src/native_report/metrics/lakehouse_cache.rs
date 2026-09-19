use std::collections::BTreeMap;

use datafusion::physical_plan::metrics::MetricsSet;

use super::labels::{labels_by_name, optional_usize, optional_usize_label, string_label, usize_label};

const NATIVE_LAKEHOUSE_CACHE_FILL_EVENT_SNAPSHOT_ROWS_METRIC_NAME: &str =
    "native_lakehouse_cache_fill_event_snapshot_rows";

/// Stable nested metric row for a background cache-fill event observed before metrics collection.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NativeLakehouseCacheFillEventMetricRow {
    /// Cache-fill outcome.
    pub outcome: String,
    /// Complete-object bytes written for successful fills, when known.
    pub bytes: Option<usize>,
    /// Elapsed nanoseconds for the background fill attempt.
    pub elapsed_ns: usize,
    /// Stable hash of the local cache path.
    pub cache_path_hash: String,
    /// Stable hash of the source stable id.
    pub stable_id_hash: String,
    /// Source file index associated with the fill, when known.
    pub file_index: Option<usize>,
    /// Source chunk index associated with the fill, when known.
    pub chunk_index: Option<usize>,
    /// Stable native error code for failed fills, when present.
    pub error_code: String,
    /// Stable native error kind for failed fills, when present.
    pub error_kind: String,
    /// Stable native error status for failed fills, when present.
    pub error_status: String,
    /// Stable native error operation for failed fills, when present.
    pub error_operation: String,
}

impl NativeLakehouseCacheFillEventMetricRow {
    /// Returns a stable key/value row used by nested report fields.
    pub fn to_stable_kv_row(&self) -> String {
        format!(
            "outcome={},bytes={},elapsed_ns={},cache_path_hash={},stable_id_hash={},file_index={},chunk_index={},error_code={},error_kind={},error_status={},error_operation={}",
            self.outcome,
            optional_usize(self.bytes),
            self.elapsed_ns,
            self.cache_path_hash,
            self.stable_id_hash,
            optional_usize(self.file_index),
            optional_usize(self.chunk_index),
            self.error_code,
            self.error_kind,
            self.error_status,
            self.error_operation,
        )
    }
}

pub(super) fn lakehouse_cache_fill_event_snapshot_rows(
    metrics: &MetricsSet,
) -> Vec<NativeLakehouseCacheFillEventMetricRow> {
    let mut rows = metrics
        .iter()
        .filter(|metric| metric.value().name() == NATIVE_LAKEHOUSE_CACHE_FILL_EVENT_SNAPSHOT_ROWS_METRIC_NAME)
        .filter(|metric| metric.labels().iter().any(|label| label.name() == "cache_path_hash"))
        .map(|metric| labels_to_lakehouse_cache_fill_event_row(labels_by_name(metric.labels())))
        .collect::<Vec<_>>();
    rows.sort_by_key(|row| row.to_stable_kv_row());
    rows
}

fn labels_to_lakehouse_cache_fill_event_row(labels: BTreeMap<&str, &str>) -> NativeLakehouseCacheFillEventMetricRow {
    NativeLakehouseCacheFillEventMetricRow {
        outcome: string_label(&labels, "outcome"),
        bytes: optional_usize_label(&labels, "bytes"),
        elapsed_ns: usize_label(&labels, "elapsed_ns"),
        cache_path_hash: string_label(&labels, "cache_path_hash"),
        stable_id_hash: string_label(&labels, "stable_id_hash"),
        file_index: optional_usize_label(&labels, "file_index"),
        chunk_index: optional_usize_label(&labels, "chunk_index"),
        error_code: string_label(&labels, "error_code"),
        error_kind: string_label(&labels, "error_kind"),
        error_status: string_label(&labels, "error_status"),
        error_operation: string_label(&labels, "error_operation"),
    }
}
